?($	k?ꍄ??y??M????s?????!?s?????$	?4?Xw6@?-N??X
@1|'?X???!\???l-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$^K?=???ŏ1w-!??A?-???1??Yd?]K???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??NbX9???A?n?????Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R?!??+??ݓ???A	?c???Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vq?-????????AbX9????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????ŏ1w-!??A????_v??Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0*??D???$??C??A?8??m4??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3?????ͪ??V??A?5^?I??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F????x???E??????A?/?$??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??e???V?/?'??A?W?2??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	"?uq??tF??_??AA??ǘ???Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
@?߾???(??y??A??<,Ԛ??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;M?O??
h"lxz??A????ׁ??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4???B?i?q??A????????Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o?ŏ1?????????A?	?c??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&I??&????j+????A??	h"??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?y?):?????&???A/n????YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??%䃞??W?/?'??A??\m????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6<?R????=yX???A?W?2??Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f????Zd;?O??AF%u???Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s??????ZӼ???A?_?L??Y?2ı.n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s?????y?&1???A???o_??Y? ?	???*	fffffr?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?D?????!?????A@)??@?????1??1_?>@:Preprocessing2F
Iterator::Model??o_??!)??杤@@)V????_??1@c?˔?4@:Preprocessing2U
Iterator::Model::ParallelMapV2 A?c?]??!$?SNC)@) A?c?]??1$?SNC)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?J?4??!?,???P@)R'??????1??p?i?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?:pΈ??!-??!@)?:pΈ??1-??!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??+e???!?8?P?+/@)X9??v??1????V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`vOj??!;:???3@)r??????1w`?M?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?? ?rh??!??SX?@)?? ?rh??1??SX?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9s	h?I?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	X?c????D1?mx???y?&1???!Zd;?O??	!       "	!       *	!       2$	??G?8T??.??i?????o_??!?_?L??:	!       B	!       J$	ýs?????????X???q??????!?2ı.n??R	!       Z$	ýs?????????X???q??????!?2ı.n??JCPU_ONLYYs	h?I?@b Y      Y@q??????B@"?
both?Your program is POTENTIALLY input-bound because 50.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.5613% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 