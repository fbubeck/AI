?($	????h????!Z???P?s???!X?5?;N??$	g?)??@D??V??@     `@!v=?q?I/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?9#J{????^)???AO??e?c??Y??	h"l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?<,Ԛ???TR'?????A?!??u???Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u????T㥛? ??A???Q???Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????9#???ʡE????A????z??Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&      ??ё\?C???AK?46??Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	????C??????A???o_??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-????????_vO??A/?$???YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?N@a????I+???A333333??Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B??o???T???A&S??:??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	X?5?;N??^K?=???A??C?l??Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
I??&???9#J{???A?j+?????Y?,C????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"?????߾??A????x???YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2???0?*???A-??????Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???3???)\???(??A]?Fx??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????MbX9??A?MbX9??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|гY????U??????Az?,C???Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(\???OjM???A?ͪ??V??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?G?z???"??~j??A9??m4???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj???*??	??A????B???Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c]?F??R???Q??AA??ǘ???Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P?s???????ɳ?A?ZB>?ټ?Y???߾??*	23333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;pΈ????!?+ԁ??C@)??N@a??1?&?m-B@:Preprocessing2F
Iterator::Model[Ӽ???!?Q??f@@)??e?c]??1??H3@:Preprocessing2U
Iterator::Model::ParallelMapV2$(~????!~)j?i*@)$(~????1~)j?i*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?t???!/ׯ?L?P@)?e?c]ܶ?1?e?Nrq%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?J?4??!?????" @)?J?4??1?????" @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateh??|?5??!"??!?U,@)?
F%u??1?Y;?e@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???z6??!?RǓx2@)8gDio??1??? ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*$????ۗ?!h???/a@)$????ۗ?1h???/a@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9w?9c@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?T)?ߕ????z???????ɳ?!^K?=???	!       "	!       *	!       2$	1?R?L:??*=??????ZB>?ټ?!?j+?????:	!       B	!       J$	????Hʙ??(e??????߾??!?D???J??R	!       Z$	????Hʙ??(e??????߾??!?D???J??JCPU_ONLYYw?9c@b Y      Y@q?/e??$A@"?
both?Your program is POTENTIALLY input-bound because 45.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.2885% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 