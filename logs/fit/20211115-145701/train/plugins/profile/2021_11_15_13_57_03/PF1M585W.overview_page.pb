?)$	???????Ry?_????S??:??!??S??[@$	?r?u?@?	?
?@ʊ?	????!LX?V??<@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$S??:???+e?X??A?c?ZB??Ye?`TR'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C?i?q??????ׁs??A?-?????YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
???W?2ı??AyX?5?;??Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}??w??/???A2U0*???Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Dio?????h??|?5??AF%u???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Dio???????x?&1??A??6???Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I??q???h??A???K7???YV????_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K???X9??v???A?\?C????YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_?????(\???A??^??Y?c]?F??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?H?}??ı.n???A?E??????Yh??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
}гY????_)?Ǻ??AΈ?????Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z????_vO??A??~j?t??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?!?uq?????H??A?V-??Y|??Pk???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????(??ޓ??Z???A?i?q????Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??S??[@_?Q???AU0*??@Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???ŏ1w-??AH?}8g??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(???????A`vOj??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_????MbX??AM?O????Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE?????	h"l??A???镲??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ݓ??@g??j+???A3ı.n???Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???_vO??[B>?٬??A,Ԛ????YEGr????*	3333???@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR'???? @!T??tF@)??j+?? @1Qj^??E@:Preprocessing2U
Iterator::Model::ParallelMapV2?HP???!?o???D@)?HP???1?o???D@:Preprocessing2F
Iterator::ModelvOjM@!EgS???G@)????z??1??GS)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8gDi?@!???JJ@)r??????1ǮhkB?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	?c???!???@)	?c???1???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateo?ŏ1??!:??	T@)??~j?t??1l}˛?p??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA??ǘ???!?S???@)Gx$(??1?$?z???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*;?O??n??!?@?'??);?O??n??1?@?'??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t48.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9Z?YJ??"@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??4H????V?z????+e?X??!g??j+???	!       "	!       *	!       2$	<?W??٦\Ă????c?ZB??!U0*??@:	!       B	!       J$	?|??<??{LG7:???%u???!7?[ A??R	!       Z$	?|??<??{LG7:???%u???!7?[ A??JCPU_ONLYYZ?YJ??"@b Y      Y@q&O@"?	
both?Your program is MODERATELY input-bound because 9.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t48.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?62.1488% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 