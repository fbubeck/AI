?($	?D????5?hc?k??333333??!c?=yX??$	?L???@Z3?a?3@R?j+@!,
?(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?W[???????St$???Ao?ŏ1??Y??z6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?HP???ё\?C???An4??@???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?*??	??U0*????A?????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??v???????&??A??o_??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&A?c?]K???ZB>????A?|?5^???Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-??????????(\???A??镲??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&S??O??e???A;M?O??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?B?i?q???C?l????A??HP??Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?=yX????N@a??A?d?`TR??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?X????L7?A`???A? ?rh???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??St$?????y???AP??n???Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????ŏ1w-??A?ǘ?????Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46??W[??????A1?*????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???߾??M?J???A?&1???Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&???Pk?w???An4??@???YHP?sע?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)????V-?????A0*??D??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???B?i??1?Zd??A??6?[??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*??????ׁs??A?G?z???Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n?????s????A<Nё\???Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2U0*?????Q????A?? ?	??YDio??ɤ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&333333??"??u????A???Mb??Yj?q?????*	33333-?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???{????!??~"ݡB@)46<???1?i?(A@:Preprocessing2F
Iterator::ModelJ+???!X?=???B@)???&S??1j? j? 6@:Preprocessing2U
Iterator::Model::ParallelMapV2?ZB>????!?䵄x/@)?ZB>????1?䵄x/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????????!??i{O@)?]K?=??1-?	??$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceaTR'????! ?!?}?@)aTR'????1 ?!?}?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateZd;?O??!L"ef?&@)x$(~???1????G@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz6?>W??!?Ј?0-@)46<???1????H@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?Zd;??!?v?k?@)?Zd;??1?v?k?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?y??d?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?]f?`???????????St$???!??N@a??	!       "	!       *	!       2$	???N?8???NX?E6?????Mb??!n4??@???:	!       B	!       J$	??T?FB????:? }????_vO??!K?=?U??R	!       Z$	??T?FB????:? }????_vO??!K?=?U??JCPU_ONLYY?y??d?@b Y      Y@q???旍<@"?
both?Your program is POTENTIALLY input-bound because 56.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?28.5531% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 